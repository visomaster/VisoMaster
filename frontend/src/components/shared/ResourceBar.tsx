import { cn } from '@/lib/utils'

interface Props {
  label: string
  value: number   // 0-100 percent
  detail?: string // e.g. "4.4 / 24 GB"
}

export function ResourceBar({ label, value, detail }: Props) {
  const color =
    value > 85 ? 'bg-red-500' :
    value > 70 ? 'bg-amber-500' :
    'bg-sky-500'

  return (
    <div className="flex items-center gap-2 min-w-0">
      <span className="text-xs text-zinc-400 shrink-0">{label}</span>
      <div className="w-20 h-1.5 bg-zinc-700 rounded-full overflow-hidden shrink-0">
        <div
          className={cn('h-full rounded-full transition-all duration-500', color)}
          style={{ width: `${Math.min(100, value)}%` }}
        />
      </div>
      <span className="text-xs text-zinc-300 shrink-0">
        {detail ?? `${Math.round(value)}%`}
      </span>
    </div>
  )
}
